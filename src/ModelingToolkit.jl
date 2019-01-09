module ModelingToolkit

using DiffEqBase
using StaticArrays, LinearAlgebra

using MacroTools
import MacroTools: splitdef, combinedef

abstract type Expression <: Number end
abstract type AbstractComponent <: Expression end
abstract type AbstractSystem end
abstract type AbstractDomain end

include("domains.jl")
include("variables.jl")

Base.promote_rule(::Type{T},::Type{T2}) where {T<:Number,T2<:Expression} = Expression
Base.one(::Type{T}) where T<:Expression = Constant(1)
Base.zero(::Type{T}) where T<:Expression = Constant(0)
Base.convert(::Type{Variable},x::Int64) = Constant(x)

function caclulate_jacobian end

@enum FunctionVersions ArrayFunction=1 SArrayFunction=2

include("operations.jl")
include("differentials.jl")
include("systems/diffeqs/diffeqsystem.jl")
include("systems/diffeqs/first_order_transform.jl")
include("systems/nonlinear/nonlinear_system.jl")
include("function_registration.jl")
include("simplify.jl")
include("utils.jl")

export Operation, Expression, AbstractComponent, AbstractDomain
export @register
end # module
