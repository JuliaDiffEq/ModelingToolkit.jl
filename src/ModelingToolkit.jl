"""
$(DocStringExtensions.README)
"""
module ModelingToolkit
using DocStringExtensions
using AbstractTrees
using DiffEqBase, SciMLBase, Reexport
using Distributed
using StaticArrays, LinearAlgebra, SparseArrays, LabelledArrays
using InteractiveUtils
using Latexify, Unitful, ArrayInterface
using MacroTools
@reexport using UnPack
using Setfield, ConstructionBase
using DiffEqJump
using DataStructures
using SpecialFunctions, NaNMath
using RuntimeGeneratedFunctions
using Base.Threads
import MacroTools: splitdef, combinedef, postwalk, striplines
import Libdl
using DocStringExtensions
using Base: RefValue
import IfElse

import Distributions

RuntimeGeneratedFunctions.init(@__MODULE__)

using RecursiveArrayTools

import SymbolicUtils
import SymbolicUtils: istree, arguments, operation, similarterm, promote_symtype,
                      Symbolic, Term, Add, Mul, Pow, Sym, FnType,
                      @rule, Rewriters, substitute
using SymbolicUtils.Code
import SymbolicUtils.Code: toexpr
import SymbolicUtils.Rewriters: Chain, Postwalk, Prewalk, Fixpoint
import JuliaFormatter

using Reexport
@reexport using Symbolics
export @derivatives
using Symbolics: _parse_vars, value, @derivatives, get_variables,
                 exprs_occur_in, solve_for, build_expr
import Symbolics: rename, get_variables!, _solve, hessian_sparsity,
                  jacobian_sparsity, islinear, _iszero, _isone,
                  tosymbol, lower_varname, diff2term, var_from_nested_derivative,
                  BuildTargets, JuliaTarget, StanTarget, CTarget, MATLABTarget,
                  ParallelForm, SerialForm, MultithreadedForm, build_function,
                  unflatten_long_ops, rhss, lhss, prettify_expr, gradient,
                  jacobian, hessian, derivative, sparsejacobian, sparsehessian,
                  substituter, scalarize

import DiffEqBase: @add_kwonly

import LightGraphs: SimpleDiGraph, add_edge!, incidence_matrix

using Requires

for fun in [:toexpr]
    @eval begin
        function $fun(eq::Equation; kw...)
            Expr(:(=), $fun(eq.lhs; kw...), $fun(eq.rhs; kw...))
        end

        $fun(eqs::AbstractArray; kw...) = map(eq->$fun(eq; kw...), eqs)
        $fun(x::Integer; kw...) = x
        $fun(x::AbstractFloat; kw...) = x
    end
end

"""
$(TYPEDEF)

TODO
"""
abstract type AbstractSystem end
abstract type AbstractTimeDependentSystem <: AbstractSystem end
abstract type AbstractTimeIndependentSystem <: AbstractSystem end
abstract type AbstractODESystem <: AbstractTimeDependentSystem end
abstract type AbstractMultivariateSystem <: AbstractSystem end

"""
$(TYPEDSIGNATURES)

Get the set of independent variables for the given system.
"""
function independent_variables end

function independent_variable end

"""
$(TYPEDSIGNATURES)

Get the set of states for the given system.
"""
function states end

"""
$(TYPEDSIGNATURES)

Get the set of parameters variables for the given system.
"""
function parameters end

include("bipartite_graph.jl")
using .BipartiteGraphs

include("variables.jl")
include("parameters.jl")

include("utils.jl")
include("domains.jl")

include("systems/abstractsystem.jl")

include("systems/diffeqs/odesystem.jl")
include("systems/diffeqs/sdesystem.jl")
include("systems/diffeqs/abstractodesystem.jl")
include("systems/diffeqs/first_order_transform.jl")
include("systems/diffeqs/modelingtoolkitize.jl")
include("systems/diffeqs/validation.jl")
include("systems/diffeqs/basic_transformations.jl")

include("systems/jumps/jumpsystem.jl")

include("systems/nonlinear/nonlinearsystem.jl")

include("systems/optimization/optimizationsystem.jl")

include("systems/control/controlsystem.jl")

include("systems/pde/pdesystem.jl")

include("systems/reaction/reactionsystem.jl")
include("systems/dependency_graphs.jl")

include("systems/discrete_system/discrete_system.jl")

include("systems/systemstructure.jl")
using .SystemStructures

include("systems/alias_elimination.jl")
include("structural_transformation/StructuralTransformations.jl")
@reexport using .StructuralTransformations

for S in subtypes(ModelingToolkit.AbstractSystem)
    S = nameof(S)
    @eval convert_system(::Type{<:$S}, sys::$S) = sys
end

struct Flow end

export AbstractTimeDependentSystem, AbstractTimeIndependentSystem, AbstractMultivariateSystem
export ODESystem, ODEFunction, ODEFunctionExpr, ODEProblemExpr, convert_system
export DAEFunctionExpr, DAEProblemExpr
export SDESystem, SDEFunction, SDEFunctionExpr, SDESystemExpr
export SystemStructure
export JumpSystem
export ODEProblem, SDEProblem
export NonlinearProblem, NonlinearProblemExpr
export OptimizationProblem, OptimizationProblemExpr
export AutoModelingToolkit
export SteadyStateProblem, SteadyStateProblemExpr
export JumpProblem, DiscreteProblem
export NonlinearSystem, OptimizationSystem
export ControlSystem
export alias_elimination, flatten, connect, @connector
export ode_order_lowering, liouville_transform
export runge_kutta_discretize
export PDESystem
export Reaction, ReactionSystem, ismassaction, oderatelaw, jumpratelaw
export Differential, expand_derivatives, @derivatives
export Equation, ConstrainedEquation
export Term, Sym
export SymScope, LocalScope, ParentScope, GlobalScope
export independent_variables, independent_variable, states, parameters, equations, controls, observed, structure
export structural_simplify
export DiscreteSystem, DiscreteProblem

export calculate_jacobian, generate_jacobian, generate_function
export calculate_control_jacobian, generate_control_jacobian
export calculate_tgrad, generate_tgrad
export calculate_gradient, generate_gradient
export calculate_factorized_W, generate_factorized_W
export calculate_hessian, generate_hessian
export calculate_massmatrix, generate_diffusion_function
export stochastic_integral_transform
export initialize_system_structure

export BipartiteGraph, equation_dependencies, variable_dependencies
export eqeq_dependencies, varvar_dependencies
export asgraph, asdigraph

export toexpr, get_variables
export simplify, substitute
export build_function
export modelingtoolkitize
export @variables, @parameters, Flow
export @named, @nonamespace, @namespace, extend, compose

end # module
