"""
$(TYPEDEF)

A scalar equation for optimization.

# Fields
$(FIELDS)

# Examples

```julia
@variables x y z
@parameters σ ρ β

op = σ*(y-x) + x*(ρ-z)-y + x*y - β*z
os = OptimizationSystem(eqs, [x,y,z],[σ,ρ,β])
```
"""
struct OptimizationSystem <: AbstractTimeIndependentSystem
    """Vector of equations defining the system."""
    op::Any
    """Unknown variables."""
    states::Vector
    """Parameters."""
    ps::Vector
    observed::Vector{Equation}
    equality_constraints::Vector{Equation}
    inequality_constraints::Vector
    """
    Name: the name of the system.  These are required to have unique names.
    """
    name::Symbol
    """
    systems: The internal systems
    """
    systems::Vector{OptimizationSystem}
    """
    defaults: The default values to use when initial conditions and/or
    parameters are not supplied in `ODEProblem`.
    """
    defaults::Dict
end

function OptimizationSystem(op, states, ps;
                            observed = [],
                            equality_constraints = Equation[],
                            inequality_constraints = [],
                            default_u0=Dict(),
                            default_p=Dict(),
                            defaults=_merge(Dict(default_u0), Dict(default_p)),
                            name = gensym(:OptimizationSystem),
                            systems = OptimizationSystem[])
    if !(isempty(default_u0) && isempty(default_p))
        Base.depwarn("`default_u0` and `default_p` are deprecated. Use `defaults` instead.", :OptimizationSystem, force=true)
    end
    sysnames = nameof.(systems)
    if length(unique(sysnames)) != length(sysnames)
        throw(ArgumentError("System names must be unique."))
    end
    defaults = todict(defaults)
    defaults = Dict(value(k) => value(v) for (k, v) in pairs(defaults))

    states, ps = value.(states), value.(ps)
    collect_defaults!(defaults, states)
    collect_defaults!(defaults, ps)

    OptimizationSystem(
                       value(op), states, ps,
                       observed,
                       equality_constraints, inequality_constraints,
                       name, systems, defaults
                      )
end

function calculate_gradient(sys::OptimizationSystem)
    expand_derivatives.(gradient(equations(sys), states(sys)))
end

function generate_gradient(sys::OptimizationSystem, vs = states(sys), ps = parameters(sys); kwargs...)
    grad = calculate_gradient(sys)
    return build_function(grad, vs, ps;
                          conv = AbstractSysToExpr(sys),kwargs...)
end

function calculate_hessian(sys::OptimizationSystem)
    expand_derivatives.(hessian(equations(sys), states(sys)))
end

function generate_hessian(sys::OptimizationSystem, vs = states(sys), ps = parameters(sys);
                          sparse = false, kwargs...)
    if sparse
        hess = sparsehessian(equations(sys),states(sys))
    else
        hess = calculate_hessian(sys)
    end
    return build_function(hess, vs, ps;
                          conv = AbstractSysToExpr(sys),kwargs...)
end

function generate_function(sys::OptimizationSystem, vs = states(sys), ps = parameters(sys); kwargs...)
    return build_function(equations(sys), vs, ps;
                          conv = AbstractSysToExpr(sys),kwargs...)
end

equations(sys::OptimizationSystem) = isempty(get_systems(sys)) ? get_op(sys) : get_op(sys) + reduce(+,namespace_expr.(get_systems(sys)))
namespace_expr(sys::OptimizationSystem) = namespace_expr(get_op(sys), nameof(sys), [])

hessian_sparsity(sys::OptimizationSystem) = hessian_sparsity(get_op(sys), states(sys))

struct AutoModelingToolkit <: DiffEqBase.AbstractADType end

DiffEqBase.OptimizationProblem(sys::OptimizationSystem,args...;kwargs...) =
    DiffEqBase.OptimizationProblem{true}(sys::OptimizationSystem,args...;kwargs...)

"""
```julia
function DiffEqBase.OptimizationProblem{iip}(sys::OptimizationSystem,
                                          parammap=DiffEqBase.NullParameters();
                                          u0=nothing, lb=nothing, ub=nothing,
                                          grad = false,
                                          hess = false, sparse = false,
                                          checkbounds = false,
                                          linenumbers = true, parallel=SerialForm(),
                                          kwargs...) where iip
```

Generates an OptimizationProblem from an OptimizationSystem and allows for automatically
symbolically calculating numerical enhancements.
"""
function DiffEqBase.OptimizationProblem{iip}(sys::OptimizationSystem, u0,
                                          parammap=DiffEqBase.NullParameters();
                                          lb=nothing, ub=nothing,
                                          grad = false,
                                          hess = false, sparse = false,
                                          checkbounds = false,
                                          linenumbers = true, parallel=SerialForm(),
                                          kwargs...) where iip
    dvs = states(sys)
    ps = parameters(sys)

    f = generate_function(sys,checkbounds=checkbounds,linenumbers=linenumbers,
                              expression=Val{false})

    if grad
        grad_oop,grad_iip = generate_gradient(sys,checkbounds=checkbounds,linenumbers=linenumbers,
                                  parallel=parallel,expression=Val{false})
        _grad(u,p) = grad_oop(u,p)
        _grad(J,u,p) = (grad_iip(J,u,p); J)
    else
        _grad = nothing
    end

    if hess
        hess_oop,hess_iip = generate_hessian(sys,checkbounds=checkbounds,linenumbers=linenumbers,
                                 sparse=sparse,parallel=parallel,expression=Val{false})
       _hess(u,p) = hess_oop(u,p)
       _hess(J,u,p) = (hess_iip(J,u,p); J)
    else
        _hess = nothing
    end

    _f = DiffEqBase.OptimizationFunction{iip,AutoModelingToolkit,typeof(f),typeof(_grad),typeof(_hess),Nothing,Nothing,Nothing,Nothing}(f,AutoModelingToolkit(),_grad,_hess,nothing,nothing,nothing,nothing)

    defs = defaults(sys)
    u0 = varmap_to_vars(u0,dvs; defaults=defs)
    p = varmap_to_vars(parammap,ps; defaults=defs)
    lb = varmap_to_vars(lb,dvs; check=false)
    ub = varmap_to_vars(ub,dvs; check=false)
    OptimizationProblem{iip}(_f,u0,p;lb=lb,ub=ub,kwargs...)
end

"""
```julia
function DiffEqBase.OptimizationProblemExpr{iip}(sys::OptimizationSystem,
                                          parammap=DiffEqBase.NullParameters();
                                          u0=nothing, lb=nothing, ub=nothing,
                                          grad = false,
                                          hes = false, sparse = false,
                                          checkbounds = false,
                                          linenumbers = true, parallel=SerialForm(),
                                          kwargs...) where iip
```

Generates a Julia expression for an OptimizationProblem from an
OptimizationSystem and allows for automatically symbolically
calculating numerical enhancements.
"""
struct OptimizationProblemExpr{iip} end

OptimizationProblemExpr(sys::OptimizationSystem,args...;kwargs...) =
    OptimizationProblemExpr{true}(sys::OptimizationSystem,args...;kwargs...)

function OptimizationProblemExpr{iip}(sys::OptimizationSystem, u0,
                                          parammap=DiffEqBase.NullParameters();
                                          lb=nothing, ub=nothing,
                                          grad = false,
                                          hess = false, sparse = false,
                                          checkbounds = false,
                                          linenumbers = false, parallel=SerialForm(),
                                          kwargs...) where iip
    dvs = states(sys)
    ps = parameters(sys)
    idx = iip ? 2 : 1
    f = generate_function(sys,checkbounds=checkbounds,linenumbers=linenumbers,
                              expression=Val{true})
    if grad
        _grad = generate_gradient(sys,checkbounds=checkbounds,linenumbers=linenumbers,
                             parallel=parallel,expression=Val{false})[idx]
    else
        _grad = :nothing
    end

    if hess
        _hess = generate_hessian(sys,checkbounds=checkbounds,linenumbers=linenumbers,
                                         sparse=sparse,parallel=parallel,expression=Val{false})[idx]
    else
        _hess = :nothing
    end

    defs = defaults(sys)
    u0 = varmap_to_vars(u0,dvs; defaults=defs)
    p = varmap_to_vars(parammap,ps; defaults=defs)
    lb = varmap_to_vars(lb,dvs)
    ub = varmap_to_vars(ub,dvs)
    quote
        f = $f
        p = $p
        u0 = $u0
        grad = $_grad
        hess = $_hess
        lb = $lb
        ub = $ub
        _f = OptimizationFunction{$iip,typeof(f),typeof(grad),typeof(hess),Nothing,Nothing,Nothing,Nothing}(f,grad,hess,nothing,AutoModelingToolkit(),nothing,nothing,nothing,0)
        OptimizationProblem{$iip}(_f,u0,p;lb=lb,ub=ub,kwargs...)
    end
end

function DiffEqBase.OptimizationFunction{iip}(f, ::AutoModelingToolkit, x, p = DiffEqBase.NullParameters();
                              grad=false, hess=false, cons = nothing, cons_j = nothing, cons_h = nothing,
                              num_cons = 0, chunksize = 1, hv = nothing) where iip

    sys = modelingtoolkitize(OptimizationProblem(f,x,p))
    u0map = states(sys) .=> x
    if p == DiffEqBase.NullParameters()
        parammap = DiffEqBase.NullParameters()
    else
        parammap = parameters(sys) .=> p
    end
    OptimizationProblem(sys,u0map,parammap,grad=grad,hess=hess).f
end
