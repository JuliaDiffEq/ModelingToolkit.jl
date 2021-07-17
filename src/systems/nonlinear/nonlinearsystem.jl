"""
$(TYPEDEF)

A nonlinear system of equations.

# Fields
$(FIELDS)

# Examples

```julia
@variables x y z
@parameters σ ρ β

eqs = [0 ~ σ*(y-x),
       0 ~ x*(ρ-z)-y,
       0 ~ x*y - β*z]
ns = NonlinearSystem(eqs, [x,y,z],[σ,ρ,β])
```
"""
struct NonlinearSystem <: AbstractTimeIndependentSystem
    """Vector of equations defining the system."""
    eqs::Vector{Equation}
    """Unknown variables."""
    states::Vector
    """Parameters."""
    ps::Vector
    observed::Vector{Equation}
    """
    Jacobian matrix. Note: this field will not be defined until
    [`calculate_jacobian`](@ref) is called on the system.
    """
    jac::RefValue{Any}
    """
    Name: the name of the system. These are required to have unique names.
    """
    name::Symbol
    """
    systems: The internal systems
    """
    systems::Vector{NonlinearSystem}
    """
    defaults: The default values to use when initial conditions and/or
    parameters are not supplied in `ODEProblem`.
    """
    defaults::Dict
    """
    structure: structural information of the system
    """
    structure::Any
    """
    type: type of the system
    """
    connection_type::Any
end

function NonlinearSystem(eqs, states, ps;
                         observed=[],
                         name=gensym(:NonlinearSystem),
                         default_u0=Dict(),
                         default_p=Dict(),
                         defaults=_merge(Dict(default_u0), Dict(default_p)),
                         systems=NonlinearSystem[],
                         connection_type=nothing,
                         )
    eqs = collect(eqs)
    if !(isempty(default_u0) && isempty(default_p))
        Base.depwarn("`default_u0` and `default_p` are deprecated. Use `defaults` instead.", :NonlinearSystem, force=true)
    end
    sysnames = nameof.(systems)
    if length(unique(sysnames)) != length(sysnames)
        throw(ArgumentError("System names must be unique."))
    end
    jac = RefValue{Any}(Matrix{Num}(undef, 0, 0))
    defaults = todict(defaults)
    defaults = Dict(value(k) => value(v) for (k, v) in pairs(defaults))

    states, ps = value.(states), value.(ps)
    collect_defaults!(defaults, states)
    collect_defaults!(defaults, ps)

    NonlinearSystem(eqs, states, ps, observed, jac, name, systems, defaults, nothing, connection_type)
end

function calculate_jacobian(sys::NonlinearSystem; sparse=false, simplify=false)
    cache = get_jac(sys)[]
    if cache isa Tuple && cache[2] == (sparse, simplify)
        return cache[1]
    end

    rhs = [eq.rhs for eq ∈ equations(sys)]
    vals = [dv for dv in states(sys)]
    if sparse
        jac = sparsejacobian(rhs, vals, simplify=simplify)
    else
        jac = jacobian(rhs, vals, simplify=simplify)
    end
    get_jac(sys)[] = jac, (sparse, simplify)
    return jac
end

function generate_jacobian(sys::NonlinearSystem, vs = states(sys), ps = parameters(sys);
                           sparse = false, simplify=false, kwargs...)
    jac = calculate_jacobian(sys,sparse=sparse, simplify=simplify)
    return build_function(jac, vs, ps;
                          conv = AbstractSysToExpr(sys), kwargs...)
end

function generate_function(sys::NonlinearSystem, dvs = states(sys), ps = parameters(sys); kwargs...)
    #obsvars = map(eq->eq.lhs, observed(sys))
    #fulldvs = [dvs; obsvars]

    rhss = [deq.rhs for deq ∈ equations(sys)]
    #rhss = Let(obss, rhss)

    return build_function(rhss, value.(dvs), value.(ps);
                          conv = AbstractSysToExpr(sys), kwargs...)
end

jacobian_sparsity(sys::NonlinearSystem) =
    jacobian_sparsity([eq.rhs for eq ∈ equations(sys)],
                      states(sys))

function DiffEqBase.NonlinearFunction(sys::NonlinearSystem, args...; kwargs...)
    NonlinearFunction{true}(sys, args...; kwargs...)
end

"""
```julia
function DiffEqBase.NonlinearFunction{iip}(sys::NonlinearSystem, dvs = states(sys),
                                     ps = parameters(sys);
                                     version = nothing,
                                     jac = false,
                                     sparse = false,
                                     kwargs...) where {iip}
```

Create an `NonlinearFunction` from the [`NonlinearSystem`](@ref). The arguments
`dvs` and `ps` are used to set the order of the dependent variable and parameter
vectors, respectively.
"""
function DiffEqBase.NonlinearFunction{iip}(sys::NonlinearSystem, dvs = states(sys),
                                     ps = parameters(sys), u0 = nothing;
                                     version = nothing,
                                     jac = false,
                                     eval_expression = true,
                                     sparse = false, simplify=false,
                                     kwargs...) where {iip}

    f_gen = generate_function(sys, dvs, ps; expression=Val{eval_expression}, kwargs...)
    f_oop,f_iip = eval_expression ? (@RuntimeGeneratedFunction(ex) for ex in f_gen) : f_gen
    f(u,p) = f_oop(u,p)
    f(du,u,p) = f_iip(du,u,p)

    if jac
        jac_gen = generate_jacobian(sys, dvs, ps;
                                    simplify=simplify, sparse = sparse,
                                    expression=Val{eval_expression}, kwargs...)
        jac_oop,jac_iip = eval_expression ? (@RuntimeGeneratedFunction(ex) for ex in jac_gen) : jac_gen
        _jac(u,p) = jac_oop(u,p)
        _jac(J,u,p) = jac_iip(J,u,p)
    else
        _jac = nothing
    end

    observedfun = let sys = sys, dict = Dict()
        function generated_observed(obsvar, u, p)
            obs = get!(dict, value(obsvar)) do
                build_explicit_observed_function(sys, obsvar)
            end
            obs(u, p)
        end
    end

    NonlinearFunction{iip}(f,
                     jac = _jac === nothing ? nothing : _jac,
                     jac_prototype = sparse ? similar(calculate_jacobian(sys, sparse=sparse),Float64) : nothing,
                     syms = Symbol.(states(sys)), observed = observedfun)
end

"""
```julia
function DiffEqBase.NonlinearFunctionExpr{iip}(sys::NonlinearSystem, dvs = states(sys),
                                     ps = parameters(sys);
                                     version = nothing,
                                     jac = false,
                                     sparse = false,
                                     kwargs...) where {iip}
```

Create a Julia expression for an `ODEFunction` from the [`ODESystem`](@ref).
The arguments `dvs` and `ps` are used to set the order of the dependent
variable and parameter vectors, respectively.
"""
struct NonlinearFunctionExpr{iip} end

function NonlinearFunctionExpr{iip}(sys::NonlinearSystem, dvs = states(sys),
                                     ps = parameters(sys), u0 = nothing;
                                     version = nothing, tgrad=false,
                                     jac = false,
                                     linenumbers = false,
                                     sparse = false, simplify=false,
                                     kwargs...) where {iip}

    idx = iip ? 2 : 1
    f = generate_function(sys, dvs, ps; expression=Val{true}, kwargs...)[idx]

    if jac
        _jac = generate_jacobian(sys, dvs, ps;
                                 sparse=sparse, simplify=simplify,
                                 expression=Val{true}, kwargs...)[idx]
    else
        _jac = :nothing
    end

    jp_expr = sparse ? :(similar($(sys.jac[]),Float64)) : :nothing

    ex = quote
        f = $f
        jac = $_jac
        NonlinearFunction{$iip}(f,
                         jac = jac,
                         jac_prototype = $jp_expr,
                         syms = $(Symbol.(states(sys))))
    end
    !linenumbers ? striplines(ex) : ex
end

function process_NonlinearProblem(constructor, sys::NonlinearSystem,u0map,parammap;
                           version = nothing,
                           jac = false,
                           checkbounds = false, sparse = false,
                           simplify=false,
                           linenumbers = true, parallel=SerialForm(),
                           eval_expression = true,
                           kwargs...)
    eqs = equations(sys)
    dvs = states(sys)
    ps = parameters(sys)
    defs = defaults(sys)
    u0 = varmap_to_vars(u0map,dvs; defaults=defs)
    p = varmap_to_vars(parammap,ps; defaults=defs)

    check_eqs_u0(eqs, dvs, u0; kwargs...)

    f = constructor(sys,dvs,ps,u0;jac=jac,checkbounds=checkbounds,
                    linenumbers=linenumbers,parallel=parallel,simplify=simplify,
                    sparse=sparse,eval_expression=eval_expression,kwargs...)
    return f, u0, p
end

function DiffEqBase.NonlinearProblem(sys::NonlinearSystem, args...; kwargs...)
    NonlinearProblem{true}(sys, args...; kwargs...)
end

"""
```julia
function DiffEqBase.NonlinearProblem{iip}(sys::NonlinearSystem,u0map,
                                          parammap=DiffEqBase.NullParameters();
                                          jac = false, sparse=false,
                                          checkbounds = false,
                                          linenumbers = true, parallel=SerialForm(),
                                          kwargs...) where iip
```

Generates an NonlinearProblem from a NonlinearSystem and allows for automatically
symbolically calculating numerical enhancements.
"""
function DiffEqBase.NonlinearProblem{iip}(sys::NonlinearSystem,u0map,
                                    parammap=DiffEqBase.NullParameters();kwargs...) where iip
    f, u0, p = process_NonlinearProblem(NonlinearFunction{iip}, sys, u0map, parammap; kwargs...)
    NonlinearProblem{iip}(f,u0,p;kwargs...)
end

"""
```julia
function DiffEqBase.NonlinearProblemExpr{iip}(sys::NonlinearSystem,u0map,
                                          parammap=DiffEqBase.NullParameters();
                                          jac = false, sparse=false,
                                          checkbounds = false,
                                          linenumbers = true, parallel=SerialForm(),
                                          kwargs...) where iip
```

Generates a Julia expression for a NonlinearProblem from a
NonlinearSystem and allows for automatically symbolically calculating
numerical enhancements.
"""
struct NonlinearProblemExpr{iip} end

function NonlinearProblemExpr(sys::NonlinearSystem, args...; kwargs...)
    NonlinearProblemExpr{true}(sys, args...; kwargs...)
end

function NonlinearProblemExpr{iip}(sys::NonlinearSystem,u0map,
                             parammap=DiffEqBase.NullParameters();
                             kwargs...) where iip

    f, u0, p = process_NonlinearProblem(NonlinearFunctionExpr{iip}, sys, u0map, parammap; kwargs...)
    linenumbers = get(kwargs, :linenumbers, true)

    ex = quote
        f = $f
        u0 = $u0
        p = $p
        NonlinearProblem(f,u0,p;$(kwargs...))
    end
    !linenumbers ? striplines(ex) : ex
end

function flatten(sys::NonlinearSystem)
    systems = get_systems(sys)
    if isempty(systems)
        return sys
    else
        return NonlinearSystem(
                               equations(sys),
                               states(sys),
                               parameters(sys),
                               observed=observed(sys),
                               defaults=defaults(sys),
                               name=nameof(sys),
                              )
    end
end

function Base.:(==)(sys1::NonlinearSystem, sys2::NonlinearSystem)
    _eq_unordered(get_eqs(sys1), get_eqs(sys2)) &&
    _eq_unordered(get_states(sys1), get_states(sys2)) &&
    _eq_unordered(get_ps(sys1), get_ps(sys2)) &&
    all(s1 == s2 for (s1, s2) in zip(get_systems(sys1), get_systems(sys2)))
end
