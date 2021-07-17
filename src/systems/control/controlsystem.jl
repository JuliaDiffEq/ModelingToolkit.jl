abstract type AbstractControlSystem <: AbstractTimeDependentSystem end

function namespace_controls(sys::AbstractControlSystem)
    [rename(x,renamespace(nameof(sys),nameof(x))) for x in controls(sys)]
end

function controls(sys::AbstractControlSystem,args...)
    name = last(args)
    extra_names = reduce(Symbol,[Symbol(:₊,nameof(x)) for x in args[1:end-1]])
    newname = renamespace(extra_names,name)
    rename(x,renamespace(nameof(sys),newname))(get_iv(sys))
end

function controls(sys::AbstractControlSystem,name::Symbol)
    x = get_controls(sys)[findfirst(x->nameof(x)==name,sys.ps)]
    rename(x,renamespace(nameof(sys),nameof(x)))
end

controls(sys::AbstractControlSystem) = isempty(get_systems(sys)) ? get_controls(sys) : [get_controls(sys);reduce(vcat,namespace_controls.(get_systems(sys)))]

"""
$(TYPEDEF)

A system describing an optimal control problem. This contains a loss function
and ordinary differential equations with control variables that describe the
dynamics.

# Fields
$(FIELDS)

# Example

```julia
using ModelingToolkit

@variables t x(t) v(t) u(t)
D = Differential(t)

loss = (4-x)^2 + 2v^2 + u^2
eqs = [
    D(x) ~ v
    D(v) ~ u^3
]

sys = ControlSystem(loss,eqs,t,[x,v],[u],[])
```
"""
struct ControlSystem <: AbstractControlSystem
    """The Loss function"""
    loss::Any
    """The ODEs defining the system."""
    eqs::Vector{Equation}
    """Independent variable."""
    iv::Sym
    """Dependent (state) variables. Must not contain the independent variable."""
    states::Vector
    """Control variables."""
    controls::Vector
    """Parameter variables. Must not contain the independent variable."""
    ps::Vector
    observed::Vector{Equation}
    """
    Name: the name of the system. These are required to have unique names.
    """
    name::Symbol
    """
    systems: The internal systems
    """
    systems::Vector{ControlSystem}
    """
    defaults: The default values to use when initial conditions and/or
    parameters are not supplied in `ODEProblem`.
    """
    defaults::Dict
    function ControlSystem(loss, deqs, iv, dvs, controls, ps, observed, name, systems, defaults)
        check_variables(dvs, iv)
        check_parameters(ps, iv)
        check_equations(deqs, iv)
        check_equations(observed, iv)
        new(loss, deqs, iv, dvs, controls, ps, observed, name, systems, defaults)
    end
end

function ControlSystem(loss, deqs::AbstractVector{<:Equation}, iv, dvs, controls, ps;
                       observed = [],
                       systems = ODESystem[],
                       default_u0=Dict(),
                       default_p=Dict(),
                       defaults=_merge(Dict(default_u0), Dict(default_p)),
                       name=gensym(:ControlSystem))
    if !(isempty(default_u0) && isempty(default_p))
        Base.depwarn("`default_u0` and `default_p` are deprecated. Use `defaults` instead.", :ControlSystem, force=true)
    end
    sysnames = nameof.(systems)
    if length(unique(sysnames)) != length(sysnames)
        throw(ArgumentError("System names must be unique."))
    end
    iv′ = value(iv)
    dvs′ = value.(dvs)
    controls′ = value.(controls)
    ps′ = value.(ps)
    defaults = todict(defaults)
    defaults = Dict(value(k) => value(v) for (k, v) in pairs(defaults))
    collect_defaults!(defaults, dvs′)
    collect_defaults!(defaults, ps′)
    ControlSystem(value(loss), deqs, iv′, dvs′, controls′,
                  ps′, observed, name, systems, defaults)
end

struct ControlToExpr
    sys::AbstractControlSystem
    states::Vector
    controls::Vector
end
ControlToExpr(@nospecialize(sys)) = ControlToExpr(sys,states(sys),controls(sys))
function (f::ControlToExpr)(O)
    !istree(O) && return O
    res = if isa(operation(O), Sym)
        # normal variables and control variables
        (any(isequal(O), f.states) || any(isequal(O), controls(f))) && return tosymbol(O)
        build_expr(:call, Any[operation(O).name; f.(arguments(O))])
    else
        build_expr(:call, Any[Symbol(operation(O)); f.(arguments(O))])
    end
end
(f::ControlToExpr)(x::Sym) = nameof(x)

function constructRadauIIA5(T::Type = Float64)
  sq6 = sqrt(convert(T, 6))
  A = [11//45-7sq6/360 37//225-169sq6/1800 -2//225+sq6/75
       37//225+169sq6/1800 11//45+7sq6/360 -2//225-sq6/75
       4//9-sq6/36 4//9+sq6/36 1//9]
  c = [2//5-sq6/10;2/5+sq6/10;1]
  α = [4//9-sq6/36;4//9+sq6/36;1//9]
  A = map(T,A)
  α = map(T,α)
  c = map(T,c)
  return DiffEqBase.ImplicitRKTableau(A,c,α,5)
end


"""
```julia
runge_kutta_discretize(sys::ControlSystem,dt,tspan;
                       tab = ModelingToolkit.constructRadauIIA5())
```

Transforms a nonlinear optimal control problem into a constrained
`OptimizationProblem` according to a Runge-Kutta tableau that describes
a collocation method. Requires a fixed `dt` over a given `timespan`.
Defaults to using the 5th order RadauIIA tableau, and altnerative tableaus
can be specified using the SciML tableau style.
"""
function runge_kutta_discretize(sys::ControlSystem,dt,tspan;
                       tab = ModelingToolkit.constructRadauIIA5())
    n = length(tspan[1]:dt:tspan[2]) - 1
    m = length(tab.α)

    sts = states(sys)
    ctr = controls(sys)
    ps = parameters(sys)
    lo = get_loss(sys)
    iv = get_iv(sys)
    f = @RuntimeGeneratedFunction(build_function([x.rhs for x in equations(sys)],sts,ctr,ps,iv,conv = ModelingToolkit.ControlToExpr(sys))[1])
    L = @RuntimeGeneratedFunction(build_function(lo,sts,ctr,ps,iv,conv = ModelingToolkit.ControlToExpr(sys)))

    var(n, i...) = var(nameof(n), i...)
    var(n::Symbol, i...) = Sym{FnType{Tuple{symtype(iv)}, Real}}(nameof(Variable(n, i...)))
    # Expand out all of the variables in time and by stages
    timed_vars = [[var(operation(x),i)(iv) for i in 1:n+1] for x in states(sys)]
    k_vars = [[var(Symbol(:ᵏ,nameof(operation(x))),i,j)(iv) for i in 1:m, j in 1:n] for x in states(sys)]
    states_timeseries = [getindex.(timed_vars,j) for j in 1:n+1]
    k_timeseries = [[Num.(getindex.(k_vars,i,j)) for i in 1:m] for j in 1:n]
    control_timeseries = [[[var(operation(x),i,j)(iv) for x in sts] for i in 1:m] for j in 1:n]
    ps = parameters(sys)
    iv = iv

    # Calculate all of the update and stage equations
    mult = [tab.A * k_timeseries[i] for i in 1:n]
    tmps = [[states_timeseries[i] .+ mult[i][j] for j in 1:m] for i in 1:n]

    bs = [states_timeseries[i] .+ dt .* reduce(+, tab.α .* k_timeseries[i],dims=1)[1] for i in 1:n]
    updates = reduce(vcat,[states_timeseries[i+1] .~ bs[i] for i in 1:n])

    df = [[dt .* Base.invokelatest(f,tmps[j][i],control_timeseries[j][i],ps,iv) for i in 1:m] for j in 1:n]
    stages = reduce(vcat,[k_timeseries[i][j] .~ df[i][j] for i in 1:n for j in 1:m])

    # Enforce equalities in the controls
    control_equality = reduce(vcat,[control_timeseries[i][end] .~ control_timeseries[i+1][1] for i in 1:n-1])

    # Create the loss function
    losses = [Base.invokelatest(L,states_timeseries[i],control_timeseries[i][1],ps,iv) for i in 1:n]
    losses = vcat(losses,[Base.invokelatest(L,states_timeseries[n+1],control_timeseries[n][end],ps,iv)])

    # Calculate final pieces
    equalities = vcat(stages,updates,control_equality)
    opt_states = vcat(reduce(vcat,reduce(vcat,states_timeseries)),reduce(vcat,reduce(vcat,k_timeseries)),reduce(vcat,reduce(vcat,control_timeseries)))

    OptimizationSystem(reduce(+,losses, init=0),opt_states,ps,equality_constraints = equalities)
end
