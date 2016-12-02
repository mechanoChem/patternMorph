/*This is the effective "main" file for the patternMorph code. All relevant classes
  are defined here. A number of the physics and numerics functions are in local
  include files. Over time all of the patternMorph code will be refactored
  Written by Krishna Garikipati & Shiva Rudraraju
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <Sacado.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <deal.II/base/tensor_function.h>
//local include files
#include "chemo1.h"
#include "chemo2.h"
#include "mechanics2.h"
#include "supplementaryFunctions1.h"
#include "projectStress.h"
#include "projectFields1.h"
#include "initialConditions1.h"
#include "boundaryConditions.h"
#include "parameters.h"
//
#include <cstdlib>
#include <ctime>
//
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/base/timer.h>
using namespace dealii;

template <int dim>
class diffusionMechanics
{
public:
  diffusionMechanics (const unsigned int mech_degree, const unsigned int diff_degree);
  ~diffusionMechanics();
  void run ();
  void solve();
  void output_results (const unsigned int cycle);
private:
  enum
    {omega1_domain_id,
     omega2_domain_id
    };

  static bool
  cell_is_in_omega1_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);
  static bool
  cell_is_in_omega2_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);
  void set_active_fe_indices ();
  void setup_dofs ();
  void assemble_system ();
  void refine_initial_grid();
  void setup_system();
  void apply_boundary_conditions();

  const unsigned int   mech_degree;
  const unsigned int   diff_degree;

  Triangulation<dim>   triangulation;
  FESystem<dim>        omega1_fe;
  FESystem<dim>        omega2_fe;
  hp::FECollection<dim> fe_collection;
  hp::DoFHandler<dim>      dof_handler;
  QGauss<dim>  	       quadrature_formula;
  QGauss<dim-1>	       face_quadrature_formula;
  SparsityPattern      sparsity_pattern;
  std::map<types::global_dof_index, double>    boundary_values;
  ConstraintMatrix hanging_node_constraints;

  //  ComputeProjection<dim> projected_stress;
  ComputeManyProjections<dim> projected_fields;

  //Matrices and vectors
  PETScWrappers::MPI::SparseMatrix system_matrix;
  PETScWrappers::MPI::Vector system_rhs;
  PETScWrappers::MPI::Vector U;
  PETScWrappers::MPI::Vector Un;
  PETScWrappers::MPI::Vector dU;
  PETScWrappers::MPI::Vector U0;
  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;

  //solution variables
  std::vector<std::string> nodal_solution_names; std::vector<DataComponentInterpretation::DataComponentInterpretation> nodal_data_component_interpretation;
  unsigned int currentIncrement, currentIteration;
  double totalTime, timeCheck1, timeCheck2, currentTime, dt;

  //parallel message stream   
  ConditionalOStream pcout;

  //For timing
  TimerOutput computing_timer;
};

// Constructor
template <int dim>
diffusionMechanics<dim>::diffusionMechanics (const unsigned int mech_degree, const unsigned int diff_degree): 
  mech_degree (mech_degree), //polynomial degree for elasticity
  diff_degree (diff_degree), //polynomial degree for transport

  //Define pdes to be solved in each domain
  omega1_fe (FE_Nothing<dim>(mech_degree), dim,
	     FE_Q<dim>(diff_degree),   4), //mu and c have same polynomial order basis functions 
  omega2_fe (FE_Nothing<dim>(mech_degree), dim,
	     FE_Q<dim>(diff_degree),   4),
  dof_handler (triangulation), quadrature_formula(3), face_quadrature_formula(2), mpi_communicator(MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)), 
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)), pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times){

  //Time stepping parameters
  totalTime = 15000; timeCheck1 = 3.0; timeCheck2 = 30;
  currentIncrement=restart_step; currentTime=restart_time;
		
  //Nodal Solution names
  for (unsigned int i=0; i<dim; ++i){
    nodal_solution_names.push_back("u"); nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
  }
  nodal_solution_names.push_back("mu1"); nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  nodal_solution_names.push_back("c1"); nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  nodal_solution_names.push_back("mu2"); nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  nodal_solution_names.push_back("c2"); nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  //FE object
  fe_collection.push_back(omega1_fe);
  fe_collection.push_back(omega2_fe);

  //  pcout.set_condition(this_mpi_process == 0);

}
template <int dim>
diffusionMechanics<dim>::~diffusionMechanics (){dof_handler.clear ();}


// Boolean functions for specifying sub-domains
template <int dim>
bool
diffusionMechanics<dim>::
cell_is_in_omega1_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell)
{
  return(cell->material_id() == omega1_domain_id);
}
template <int dim>
bool
diffusionMechanics<dim>::
cell_is_in_omega2_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell)
{
  return(cell->material_id() == omega2_domain_id);
}

// Set active fe indices in each sub-domain
template <int dim>
void
diffusionMechanics<dim>::set_active_fe_indices()
{
  for (typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
       cell != dof_handler.end(); ++cell)
    {
      if (cell_is_in_omega1_domain(cell))
	cell->set_active_fe_index(0);
      else if (cell_is_in_omega2_domain(cell))
	cell->set_active_fe_index(1);
      else
	Assert(false, ExcNotImplemented());
    }
}

// Call function to set active indices and distribute dofs
template <int dim>
void
diffusionMechanics<dim>::setup_dofs()
{
  set_active_fe_indices ();
  dof_handler.distribute_dofs(fe_collection);
}

// Assemble system
template <int dim>
void diffusionMechanics<dim>::assemble_system(){
  system_matrix=0; system_rhs=0; boundary_values.clear();
  hp::QCollection<dim> hp_q_collection;
  hp_q_collection.push_back(quadrature_formula);
  hp_q_collection.push_back(quadrature_formula);
  hp::FEValues<dim> hp_fe_values (fe_collection, hp_q_collection, update_values | update_gradients | update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> omega1_fe_face_values (omega1_fe, face_quadrature_formula, update_values | update_quadrature_points | update_JxW_values | update_normal_vectors);
  FEFaceValues<dim> omega2_fe_face_values (omega2_fe, face_quadrature_formula, update_values | update_quadrature_points | update_JxW_values | update_normal_vectors);
  const unsigned int omega1_dofs_per_cell = omega1_fe.dofs_per_cell;
  const unsigned int omega2_dofs_per_cell = omega2_fe.dofs_per_cell;
  FullMatrix<double>   local_matrix;
  Vector<double>       local_rhs;
 
  //loop over cells  
  typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end(); 
  //AD variables
  PETScWrappers::Vector localized_U(U);
  PETScWrappers::Vector localized_Un(Un);
  PETScWrappers::Vector localized_U0(U0);
  unsigned ncell = 0;
  for (;cell!=endc; ++cell){
    if (cell->subdomain_id() == this_mpi_process)
      {
	hp_fe_values.reinit(cell);
	const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
	local_matrix.reinit (cell->get_fe().dofs_per_cell,
			     cell->get_fe().dofs_per_cell);
	local_rhs.reinit (cell->get_fe().dofs_per_cell);
	if (cell_is_in_omega1_domain(cell))
	  {
	    const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
	    Assert (dofs_per_cell == omega1_dofs_per_cell, ExcInternalError());
	    FEFaceValues<dim> & fe_face_values = omega1_fe_face_values;
	    
	    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
	    unsigned int n_q_points= fe_values.n_quadrature_points;

	    local_matrix = 0; local_rhs = 0; 
	    cell->get_dof_indices (local_dof_indices);	

	    //Local dof vectors: Some are Sacado variables for automatic differentiation
	    Table<1, Sacado::Fad::DFad<double> > ULocal(dofs_per_cell); Table<1, double > ULocalConv(dofs_per_cell); Table<1, double > U0Local(dofs_per_cell);
	    for (unsigned int i=0; i<dofs_per_cell; ++i)
	      {
		if (std::abs(localized_U(local_dof_indices[i]))<1.0e-16){ULocal[i]=0.0;}
		else{ULocal[i]=localized_U(local_dof_indices[i]);}
		ULocal[i].diff (i, dofs_per_cell);
		ULocalConv[i]= localized_Un(local_dof_indices[i]);
		U0Local[i]= localized_U0(local_dof_indices[i]);
	      }

	    //More Sacado variables for automatic differentiation, and some doubles
	    deformationMap<Sacado::Fad::DFad<double>, dim> defMap(n_q_points); 
	    getDeformationMap<Sacado::Fad::DFad<double>, dim>(fe_values, 0, ULocal, defMap, currentIteration);
	    Table<1, Sacado::Fad::DFad<double> > R(dofs_per_cell);
	    for (unsigned int i = 0; i < dofs_per_cell; ++i){R[i] = 0.0;}
	    dealii::Table<1,double> mu1_conv(n_q_points);
	    dealii::Table<1,Sacado::Fad::DFad<double> > mu1(n_q_points), mu1_0(n_q_points);
	    dealii::Table<2,Sacado::Fad::DFad<double> > gradmu1(n_q_points, dim);
	    dealii::Table<1,double> c1_conv(n_q_points);
	    dealii::Table<1,Sacado::Fad::DFad<double> > c1(n_q_points), c1_0(n_q_points);
	    dealii::Table<2,Sacado::Fad::DFad<double> > gradc1(n_q_points, dim);
	    dealii::Table<1,double> mu2_conv(n_q_points);
	    dealii::Table<1,Sacado::Fad::DFad<double> > mu2(n_q_points), mu2_0(n_q_points);
	    dealii::Table<2,Sacado::Fad::DFad<double> > gradmu2(n_q_points, dim);
	    dealii::Table<1,double> c2_conv(n_q_points);
	    dealii::Table<1,Sacado::Fad::DFad<double> > c2(n_q_points), c2_0(n_q_points);
	    dealii::Table<2,Sacado::Fad::DFad<double> > gradc2(n_q_points, dim);

	    //Chem potential and concentration, and their gradients	    
	    for (unsigned int q=0; q<n_q_points; ++q)
	      {
		mu1_conv[q] = 0.0; c1_conv[q]=0.0; mu2_conv[q] = 0.0; c2_conv[q]=0.0;
		mu1[q] = 0.0; c1[q]=0.0; mu2[q] = 0.0; c2[q]=0.0;
		mu1_0[q] = 0.0; c1_0[q]=0.0; mu2_0[q] = 0.0; c2_0[q]=0.0;
		for (unsigned int j = 0; j < dim; ++j)
		  {
		    gradmu1[q][j] = 0.0;
		    gradc1[q][j] = 0.0;
		    gradmu2[q][j] = 0.0;
		    gradc2[q][j] = 0.0;
		  }
		for (unsigned int i = 0; i < dofs_per_cell; ++i) 
		  {
		    const unsigned int ck = fe_values.get_fe().system_to_component_index(i).first - dim;
		    if (ck==0) 
		      {
			mu1[q]+=fe_values.shape_value(i, q)*ULocal[i]; 
			mu1_conv[q]+=fe_values.shape_value(i, q)*ULocalConv[i]; 
			mu1_0[q]+=fe_values.shape_value(i, q)*U0Local[i];
			for (unsigned int j = 0; j < dim; ++j)
			  {
			    gradmu1[q][j] += fe_values.shape_grad(i, q)[j]*ULocal[i];
			  }
		      }
		    else if (ck==1) 
		      {
			c1[q]+=fe_values.shape_value(i, q)*ULocal[i]; 
			c1_conv[q]+=fe_values.shape_value(i, q)*ULocalConv[i]; 
			c1_0[q]+=fe_values.shape_value(i, q)*U0Local[i];
			for (unsigned int j = 0; j < dim; ++j)
			  {
			    gradc1[q][j] += fe_values.shape_grad(i, q)[j]*ULocal[i];
			  }
		      }
		    else if (ck==2) 
		      {
			mu2[q]+=fe_values.shape_value(i, q)*ULocal[i]; 
			mu2_conv[q]+=fe_values.shape_value(i, q)*ULocalConv[i]; 
			mu2_0[q]+=fe_values.shape_value(i, q)*U0Local[i];
			for (unsigned int j = 0; j < dim; ++j)
			  {
			    gradmu2[q][j] += fe_values.shape_grad(i, q)[j]*ULocal[i];
			  }
		      }
		    else if (ck==3) 
		      {
			c2[q]+=fe_values.shape_value(i, q)*ULocal[i]; 
			c2_conv[q]+=fe_values.shape_value(i, q)*ULocalConv[i]; 
			c2_0[q]+=fe_values.shape_value(i, q)*U0Local[i];
			for (unsigned int j = 0; j < dim; ++j)
			  {
			    gradc2[q][j] += fe_values.shape_grad(i, q)[j]*ULocal[i];
			  }
		      }
		  }
	      }

	    //Elasticity and transport residuals
	    residualForMechanics<dim>(fe_values, 0, fe_face_values, ULocal, ULocalConv, R, defMap, cell, c1_conv, c1, c1_0, currentTime, totalTime);
	    residualForChemo1<dim>(fe_values, dim, fe_face_values, cell, dt, ULocal, ULocalConv, R, defMap, mu1_conv, mu1, gradmu1, c1_conv, c1, c2, gradc1); //mu1, c1
	    residualForChemo2<dim>(fe_values, dim, fe_face_values, cell, dt, ULocal, ULocalConv, R, defMap, mu2_conv, mu2, gradmu2, c2_conv, c1, c2, gradc2); //mu2, c2	    
		
	    //Residual(R) and Jacobian(R')
	    for (unsigned int i=0; i<dofs_per_cell; ++i) 
	      {
		for (unsigned int j=0; j<dofs_per_cell; ++j)
		  {
		    // R' by AD
		    local_matrix(i,j)= R[i].dx(j);
		  }
		//R
		local_rhs(i) = -R[i].val();
	      }
		
	    //Global Assembly
	    hanging_node_constraints.distribute_local_to_global(local_matrix, local_rhs,
								local_dof_indices,
								system_matrix, system_rhs);
	  }
	else //cell_is_in_omega2_domain(cell)
	  {
	    const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
	    Assert (dofs_per_cell == omega2_dofs_per_cell, ExcInternalError());
	    FEFaceValues<dim> & fe_face_values = omega2_fe_face_values;
	    
	    std::vector<unsigned int> local_dof_indices (dofs_per_cell);
	    unsigned int n_q_points= fe_values.n_quadrature_points;
	    
	    local_matrix = 0; local_rhs = 0; 
	    cell->get_dof_indices (local_dof_indices);	

	    //Local dof vectors: Some are Sacado variables for automatic differentiation
	    Table<1, Sacado::Fad::DFad<double> > ULocal(dofs_per_cell); Table<1, double > ULocalConv(dofs_per_cell); Table<1, double > U0Local(dofs_per_cell);
	    for (unsigned int i=0; i<dofs_per_cell; ++i)
	      {
		if (std::abs(localized_U(local_dof_indices[i]))<1.0e-16){ULocal[i]=0.0;}
		else{ULocal[i]=localized_U(local_dof_indices[i]);}
		ULocal[i].diff (i, dofs_per_cell);
		ULocalConv[i]= localized_Un(local_dof_indices[i]);
		U0Local[i]= localized_U0(local_dof_indices[i]);
	      }

	    //More Sacado variables for automatic differentiation, and some doubles
	    deformationMap<Sacado::Fad::DFad<double>, dim> defMap(n_q_points); 
	    getDeformationMap<Sacado::Fad::DFad<double>, dim>(fe_values, 0, ULocal, defMap, currentIteration);
	    Table<1, Sacado::Fad::DFad<double> > R(dofs_per_cell);
	    for (unsigned int i = 0; i < dofs_per_cell; ++i){R[i] = 0.0;}
	    dealii::Table<1,double> mu1_conv(n_q_points);
	    dealii::Table<1,Sacado::Fad::DFad<double> > mu1(n_q_points), mu1_0(n_q_points);
	    dealii::Table<2,Sacado::Fad::DFad<double> > gradmu1(n_q_points, dim);
	    dealii::Table<1,double> c1_conv(n_q_points);
	    dealii::Table<1,Sacado::Fad::DFad<double> > c1(n_q_points), c1_0(n_q_points);
	    dealii::Table<2,Sacado::Fad::DFad<double> > gradc1(n_q_points, dim);
	    dealii::Table<1,double> mu2_conv(n_q_points);
	    dealii::Table<1,Sacado::Fad::DFad<double> > mu2(n_q_points), mu2_0(n_q_points);
	    dealii::Table<2,Sacado::Fad::DFad<double> > gradmu2(n_q_points, dim);
	    dealii::Table<1,double> c2_conv(n_q_points);
	    dealii::Table<1,Sacado::Fad::DFad<double> > c2(n_q_points), c2_0(n_q_points);
	    dealii::Table<2,Sacado::Fad::DFad<double> > gradc2(n_q_points, dim);

	    //Chem potential and concentration, and their gradients	    	    
	    for (unsigned int q=0; q<n_q_points; ++q)
	      {
		mu1_conv[q] = 0.0; c1_conv[q]=0.0; mu2_conv[q] = 0.0; c2_conv[q]=0.0;
		mu1[q] = 0.0; c1[q]=0.0; mu2[q] = 0.0; c2[q]=0.0;
		mu1_0[q] = 0.0; c1_0[q]=0.0; mu2_0[q] = 0.0; c2_0[q]=0.0;
		for (unsigned int j = 0; j < dim; ++j)
		  {
		    gradmu1[q][j] = 0.0;
		    gradc1[q][j] = 0.0;
		    gradmu2[q][j] = 0.0;
		    gradc2[q][j] = 0.0;
		  }
		for (unsigned int i = 0; i < dofs_per_cell; ++i) 
		  {
		    const unsigned int ck = fe_values.get_fe().system_to_component_index(i).first - dim;
		    if (ck==0) 
		      {
			mu1[q]+=fe_values.shape_value(i, q)*ULocal[i]; 
			mu1_conv[q]+=fe_values.shape_value(i, q)*ULocalConv[i]; 
			mu1_0[q]+=fe_values.shape_value(i, q)*U0Local[i];
			for (unsigned int j = 0; j < dim; ++j)
			  {
			    gradmu1[q][j] += fe_values.shape_grad(i, q)[j]*ULocal[i];
			  }
		      }
		    else if (ck==1) 
		      {
			c1[q]+=fe_values.shape_value(i, q)*ULocal[i]; 
			c1_conv[q]+=fe_values.shape_value(i, q)*ULocalConv[i]; 
			c1_0[q]+=fe_values.shape_value(i, q)*U0Local[i];
			for (unsigned int j = 0; j < dim; ++j)
			  {
			    gradc1[q][j] += fe_values.shape_grad(i, q)[j]*ULocal[i];
			  }
		      }
		    else if (ck==2) 
		      {
			mu2[q]+=fe_values.shape_value(i, q)*ULocal[i]; 
			mu2_conv[q]+=fe_values.shape_value(i, q)*ULocalConv[i]; 
			mu2_0[q]+=fe_values.shape_value(i, q)*U0Local[i];
			for (unsigned int j = 0; j < dim; ++j)
			  {
			    gradmu2[q][j] += fe_values.shape_grad(i, q)[j]*ULocal[i];
			  }
		      }
		    else if (ck==3) 
		      {
			c2[q]+=fe_values.shape_value(i, q)*ULocal[i]; 
			c2_conv[q]+=fe_values.shape_value(i, q)*ULocalConv[i]; 
			c2_0[q]+=fe_values.shape_value(i, q)*U0Local[i];
			for (unsigned int j = 0; j < dim; ++j)
			  {
			    gradc2[q][j] += fe_values.shape_grad(i, q)[j]*ULocal[i];
			  }
		      }
		  }
	      }

	    //Elasticity and transport residuals
	    residualForMechanics<dim>(fe_values, 0, fe_face_values, ULocal, ULocalConv, R, defMap, cell, c1_conv, c1, c1_0, currentTime, totalTime);
	    residualForChemo1<dim>(fe_values, dim, fe_face_values, cell, dt, ULocal, ULocalConv, R, defMap, mu1_conv, mu1, gradmu1, c1_conv, c1, c2, gradc1); //mu1, c1
	    residualForChemo2<dim>(fe_values, dim, fe_face_values, cell, dt, ULocal, ULocalConv, R, defMap, mu2_conv, mu2, gradmu2, c2_conv, c1, c2, gradc2); //mu2, c2	    
		
	    //Residual(R) and Jacobian(R')
	    for (unsigned int i=0; i<dofs_per_cell; ++i)//Does not include diffusion dofs if FE_Nothing used for omega2_domain
	      {
		for (unsigned int j=0; j<dofs_per_cell; ++j)//Does not include diffusion dofs if FE_Nothing used for omega2_domain
		  {
		    // R' by AD
		    local_matrix(i,j)= R[i].dx(j);
		  }
	      //R
		local_rhs(i) = -R[i].val();
	      }
		
	  //Global Assembly
	    hanging_node_constraints.distribute_local_to_global(local_matrix, local_rhs,
								local_dof_indices,
								system_matrix, system_rhs);
	  }
      }
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);	
  //Applying Dirichlet BCs: Displacement == 0 on x = 0, concentration = 0 on r = inner_radius
  /*std::vector<bool> boundary_x0 (totalDOF, false); boundary_x0[totalDOF-5]=true; //boundary_x0[totalDOF-4]=true; boundary_x0[totalDOF-3]=true; 
    std::vector<bool> boundary_ri (totalDOF, false); boundary_ri[totalDOF-5]=true; boundary_ri[totalDOF-4]=true; boundary_ri[totalDOF-3]=true; boundary_ri[totalDOF-2]=true;*/

  //Applying Dirichlet BCS: Normal displacement == 0 on x = 0, a; y = 0, b; all displacements == 0 on z = 0
  //Applying Dirichlet BCS for transport problem: Homogeneous Neumann on all boundaries except x == a, which has Dirichlet BCs
  std::vector<bool> boundary_x0 (totalDOF, false); boundary_x0[totalDOF-7]=true;
  std::vector<bool> boundary_xa (totalDOF, false); boundary_xa[totalDOF-7]=true; //boundary_xa[totalDOF-4]=true; boundary_xa[totalDOF-2]=true; //Dirichlet bcs on composition via potential
  std::vector<bool> boundary_y0 (totalDOF, false); boundary_y0[totalDOF-6]=true; 
  std::vector<bool> boundary_yb (totalDOF, false); boundary_yb[totalDOF-6]=true; 
  std::vector<bool> boundary_z0 (totalDOF, false); boundary_z0[totalDOF-7]=true; boundary_z0[totalDOF-6]=true; boundary_z0[totalDOF-5]=true;
 
  if (currentIteration == 0)
    {
      //For half shell
      /*      VectorTools:: interpolate_boundary_values (dof_handler, 1, ZeroFunction<dim> (totalDOF), boundary_values, boundary_x0); // on x=0 boundary
	      VectorTools:: interpolate_boundary_values (dof_handler, 0, BoundaryConditions<dim> (currentTime), boundary_values, boundary_ri); // on r=inner_radius*/
      //For rectangular domain
      VectorTools:: interpolate_boundary_values (dof_handler, 1, ZeroFunction<dim> (totalDOF), boundary_values, boundary_x0); // on x=0 boundary
      //      VectorTools:: interpolate_boundary_values (dof_handler, 4, BoundaryConditions<dim> (currentTime), boundary_values, boundary_xa);//Dirichlet bcs on x=a boundary 
      VectorTools:: interpolate_boundary_values (dof_handler, 4, ZeroFunction<dim> (totalDOF), boundary_values, boundary_xa);//on x=a boundary
      VectorTools:: interpolate_boundary_values (dof_handler, 2, ZeroFunction<dim> (totalDOF), boundary_values, boundary_y0); // on y=0 boundary
      VectorTools:: interpolate_boundary_values (dof_handler, 5, ZeroFunction<dim> (totalDOF), boundary_values, boundary_yb); // on y=b boundary
      VectorTools:: interpolate_boundary_values (dof_handler, 3, ZeroFunction<dim> (totalDOF), boundary_values, boundary_z0); // on z=0 boundary      
    }
  
  else
    {
      //For half shell
      /*      VectorTools:: interpolate_boundary_values (dof_handler, 1, ZeroFunction<dim> (totalDOF), boundary_values, boundary_x0); // on x=0 boundary
	      VectorTools:: interpolate_boundary_values (dof_handler, 0, ZeroFunction<dim> (totalDOF), boundary_values, boundary_ri); // on r=inner_radius*/
      //For rectangular domain
      VectorTools:: interpolate_boundary_values (dof_handler, 1, ZeroFunction<dim> (totalDOF), boundary_values, boundary_x0); // on x=0 boundary
      VectorTools:: interpolate_boundary_values (dof_handler, 4, ZeroFunction<dim> (totalDOF), boundary_values, boundary_xa); // on x=a boundary
      VectorTools:: interpolate_boundary_values (dof_handler, 2, ZeroFunction<dim> (totalDOF), boundary_values, boundary_y0); // on y=0 boundary
      VectorTools:: interpolate_boundary_values (dof_handler, 5, ZeroFunction<dim> (totalDOF), boundary_values, boundary_yb); // on y=b boundary
      VectorTools:: interpolate_boundary_values (dof_handler, 3, ZeroFunction<dim> (totalDOF), boundary_values, boundary_z0); // on z=0 boundary
    }

  //End of Applying Dirichlet BC
  dU=0;
  MatrixTools::apply_boundary_values (boundary_values, system_matrix, dU, system_rhs, false);
}


//Solve
template <int dim>
void diffusionMechanics<dim>::solve(){
  double res=1, tol=1.0e-16, abs_tol=1.0e-8, initial_norm=0, current_norm=0;
  double machineEPS=1.0e-15;
  currentIteration=0;
  pcout << std::endl;
  while (true){
    if (currentIteration>=10) {PetscPrintf (mpi_communicator,"Maximum number of iterations reached without convergence. \n"); break; exit (1);}
    if (currentIteration > 0 && current_norm <= abs_tol) {PetscPrintf (mpi_communicator,"Convergence in absolute norm \n"); break; exit (1);}

    computing_timer.enter_section("assembly");
    assemble_system();
    computing_timer.exit_section("assembly");
    current_norm=system_rhs.l2_norm();    
    initial_norm=std::max(initial_norm, current_norm);
    res=current_norm/initial_norm;
    PetscPrintf (mpi_communicator, "Inc:%3u (time:%10.4e, dt:%10.4e), Iter:%2u. Residual norm: %10.2e. Relative norm: %10.2e \n", currentIncrement, currentTime, dt,  currentIteration, current_norm, res);
    if (res<tol || current_norm< abs_tol){PetscPrintf (mpi_communicator, "Residual converged in %u iterations.\n\n", currentIteration); break;}
    
    PetscPrintf(mpi_communicator, "Solving... \n");

    //Parallel direct solver
    computing_timer.enter_section("solve");

    SolverControl solver_control;
    PETScWrappers::SolverPreOnly solver(solver_control, mpi_communicator);
    PETScWrappers::PreconditionLU preconditioner(system_matrix);
    solver.solve(system_matrix, dU, system_rhs, preconditioner);
    computing_timer.exit_section("solve");
    PetscPrintf(mpi_communicator, "Solved! \n");
    PETScWrappers::Vector localized_dU (dU);
    hanging_node_constraints.distribute (localized_dU);
    dU = localized_dU;

    U+=dU;
    ++currentIteration;
    pcout << std::flush;
  }
  Un=U;
}

//Mark boundaries

template <int dim>
void diffusionMechanics<dim>::apply_boundary_conditions(){
  typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
  for (;cell!=endc; ++cell){
    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f){
      if (cell->face(f)->at_boundary()){
	cell->face(f)->set_boundary_indicator (0);
	const Point<dim> face_center = cell->face(f)->center();
	/*	if (face_center.distance(center) >= 0.95*outer_radius){
	  cell->face(f)->set_boundary_indicator (2); //outer_radius
	  }*/
	if (face_center[0] == 0.0 ){
	  cell->face(f)->set_boundary_indicator (1); //X
	}		
	if (face_center[1] == 0.0){
	  cell->face(f)->set_boundary_indicator (2); //Y
	}		
	if (face_center[2] == 0.0){
	  cell->face(f)->set_boundary_indicator (3); //Z
	}
	if (face_center[0] == alen){
	  cell->face(f)->set_boundary_indicator (4); //X
	}
	if (face_center[1] == blen){
	  cell->face(f)->set_boundary_indicator (5); //Y
	}
	if (face_center[2] == clen){
	  cell->face(f)->set_boundary_indicator (6); //Z
	}
      }
    }
  }
}

//Setup
template <int dim>
void diffusionMechanics<dim>::setup_system(){
  GridTools::partition_triangulation (n_mpi_processes, triangulation);
  dof_handler.distribute_dofs (fe_collection);
  DoFRenumbering::subdomain_wise(dof_handler);
  const types::global_dof_index n_local_dofs = DoFTools::count_dofs_with_subdomain_association(dof_handler, this_mpi_process);
  system_matrix.reinit (mpi_communicator,
			dof_handler.n_dofs(), 
			dof_handler.n_dofs(), 
			n_local_dofs,
			n_local_dofs,
			dof_handler.max_couplings_between_dofs());
  U.reinit (mpi_communicator,dof_handler.n_dofs(),n_local_dofs); 
  Un.reinit (mpi_communicator,dof_handler.n_dofs(),n_local_dofs);  
  dU.reinit (mpi_communicator,dof_handler.n_dofs(),n_local_dofs); 
  U0.reinit (mpi_communicator,dof_handler.n_dofs(),n_local_dofs);
  system_rhs.reinit (mpi_communicator,dof_handler.n_dofs(),n_local_dofs); 

  hanging_node_constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
					   hanging_node_constraints);
  hanging_node_constraints.close ();

  pcout << "   Number of active cells:       " << triangulation.n_active_cells() << std::endl;
  pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl; 
}

//Initial zoned mesh
template <int dim>
void diffusionMechanics<dim>::refine_initial_grid (){
  pcout << "Initial refinement in progress" << std::endl;
  bool furtherRefine=true;
  double finestCellWidth = 0.15*(outer_radius-inner_radius);
  pcout << "Finest cell width= " << finestCellWidth << std::endl;
  while (furtherRefine){
    furtherRefine=false;
    typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
    for (;cell!=endc; ++cell){
      if (std::sqrt(cell->center().square()) >= 0.2*inner_radius + 0.8*outer_radius){
	double cellWidth=std::pow(cell->measure(), 1.0/dim);
	if (cellWidth>finestCellWidth){
	  cell->set_refine_flag();
	  if (cellWidth>3*finestCellWidth){
	  furtherRefine=true;
	  }
	}
      }
    }
  triangulation.execute_coarsening_and_refinement ();
  }
  pcout << "Initial refinement DONE" << std::endl;
}


//Output results
template <int dim>
void diffusionMechanics<dim>::output_results (const unsigned int cycle){
  const PETScWrappers::Vector localized_U(U);

  //Write results to VTK file
  if (this_mpi_process == 0)
    {     
      std::ostringstream filename1; filename1 << "layer-ch3well-2field-dt-large-long" << cycle << ".vtk"; std::ofstream output1 (filename1.str().c_str());
      DataOut<dim, hp::DoFHandler<dim> > data_out; data_out.attach_dof_handler (dof_handler);

      //Add nodal DOF data
      data_out.add_data_vector (localized_U, nodal_solution_names, DataOut<dim, hp::DoFHandler<dim> >::type_dof_data, nodal_data_component_interpretation);
      data_out.add_data_vector (localized_U, projected_fields);
      std::vector<unsigned int> partition_int (triangulation.n_active_cells());
      GridTools::get_subdomain_association (triangulation, partition_int);
      const Vector<double> partitioning(partition_int.begin(), partition_int.end());
      data_out.add_data_vector (partitioning, "partitioning");
      data_out.build_patches (); data_out.write_vtk (output1); output1.close();
    }
}

//Run
template <int dim>
void diffusionMechanics<dim>::run (){

  
  std::vector<std::vector <double> >  step_sizes;
  std::vector <double> step0;
  for (unsigned int j = 0; j < 150; ++j)
    step0.push_back(alen/150.0); 
  std::vector <double> step1;
  for (unsigned int j = 0; j < 150; ++j)
    step1.push_back(blen/150.0);
  std::vector <double> step2;
  for (unsigned int j = 0; j < 8; ++j)
    {
      step2.push_back(0.8*clen/8.0);
    }
  for (unsigned int j = 0; j < 2; ++j)
    {
      step2.push_back(0.2*clen/2.0);
    }
  step_sizes.push_back(step0); step_sizes.push_back(step1); step_sizes.push_back(step2); 

  GridGenerator::subdivided_hyper_rectangle (triangulation, step_sizes,
                                             Point<3>(0.0,0.0,0.0),
                                             Point<3>(alen,blen,clen),false);
  
  
  /*  GridGenerator::half_hyper_shell(triangulation, center, inner_radius, outer_radius, n_cells = 0, false);
  const HalfHyperShellBoundary<dim> boundary_description(center, inner_radius, outer_radius);
  triangulation.set_boundary (0, boundary_description);
  triangulation.refine_global (3);
  refine_initial_grid();*/
  //Mark cells by sub-domain
  Point<dim> cellcenter (0.0,0.0,0.0);
  for (typename Triangulation<dim>::active_cell_iterator cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell){
    cellcenter = cell->center();
    //    if (cellcenter.distance(center) >= 2.5*inner_radius)
    if (cellcenter[dim-1] >= 0.8*clen)
      cell->set_material_id(omega1_domain_id);
    else
      cell->set_material_id(omega2_domain_id);
  }
  setup_dofs();
  setup_system(); 
  //Now solving
  srand (static_cast <unsigned> (time(0)));//Seed random initial conditions
  VectorTools::interpolate(dof_handler, InitialConditions<dim>(), U0); 

  apply_boundary_conditions();

  if (restart_step == 0)
    {
      U=U0; Un=U0;
      output_results(0); //output initial state
    }
  else
    {
      std::ostringstream filename2;
      //      filename2 << "restart-" << restart_step << ".U";
      //      PetscPrintf(mpi_communicator, "\nReading solution from restart files: %s \n", filename2);
      //Read solution vector
      //      std::ifstream input2 (filename2.str().c_str());
      //      Un.block_read(input2);
      U = Un;
    }

  //Initial time stepping
  currentIncrement=restart_step;
  dt = 0.1;//Initial time step
  for (currentTime=restart_time; currentTime < timeCheck1; currentTime+=dt){
    currentIncrement++;
    solve(); 
    output_results(currentIncrement); //output solution at current load increment
  }

  //Modified time stepping
  dt = 1.0;//Modified time step
  for (currentTime=timeCheck1; currentTime < timeCheck2; currentTime+=dt){
    currentIncrement++;
    solve(); 
    output_results(currentIncrement); //output solution at current load increment
  }

  //Modified time stepping
  dt = 2.0;//Modified time step
  for (currentTime=timeCheck2; currentTime <= totalTime; currentTime+=dt){
    currentIncrement++;
    solve(); 
    output_results(currentIncrement); //output solution at current load increment
  }
}

//Main
int main (int argc, char **argv){
  try{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    deallog.depth_console (0);
    diffusionMechanics<DIMS> problem(1,1);
    problem.run ();
  }
  catch (std::exception &exc){
    std::cerr << std::endl << std::endl
	      << "----------------------------------------------------"
	      << std::endl;
    std::cerr << "Exception on processing: " << std::endl
	      << exc.what() << std::endl
	      << "Aborting!" << std::endl
	      << "----------------------------------------------------"
	      << std::endl;

    return 1;
  }
  catch (...){
    std::cerr << std::endl << std::endl
	      << "----------------------------------------------------"
	      << std::endl;
    std::cerr << "Unknown exception!" << std::endl
	      << "Aborting!" << std::endl
	      << "----------------------------------------------------"
	      << std::endl;
    return 1;
  }

  return 0;
}
