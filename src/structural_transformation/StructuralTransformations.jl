module StructuralTransformations

const UNVISITED = typemin(Int)
const UNASSIGNED = typemin(Int)

using Setfield: @set!, @set
using UnPack: @unpack

using SymbolicUtils
using SymbolicUtils.Code
using SymbolicUtils.Rewriters
using SymbolicUtils: similarterm, istree

using ModelingToolkit
using ModelingToolkit: ODESystem, AbstractSystem,var_from_nested_derivative, Differential,
                       states, equations, vars, Symbolic, diff2term, value,
                       operation, arguments, Sym, Term, simplify, solve_for,
                       isdiffeq, isdifferential, get_structure, get_iv, independent_variables,
                       get_structure, defaults, InvalidSystemException,
                       ExtraEquationsSystemException,
                       ExtraVariablesSystemException

using ModelingToolkit.BipartiteGraphs
using LightGraphs
using ModelingToolkit.SystemStructures

using ModelingToolkit.DiffEqBase
using ModelingToolkit.StaticArrays
using ModelingToolkit: @RuntimeGeneratedFunction, RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

using SparseArrays

using NonlinearSolve

export tearing, dae_index_lowering, check_consistency
export build_torn_function, build_observed_function, ODAEProblem
export sorted_incidence_matrix

include("utils.jl")
include("pantelides.jl")
include("bipartite_tearing/modia_tearing.jl")
include("tearing.jl")
include("codegen.jl")

end # module
